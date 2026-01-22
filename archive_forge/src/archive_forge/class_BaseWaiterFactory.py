from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
class BaseWaiterFactory:
    """
    A helper class used for creating additional waiters.
    Unlike the waiters available directly from botocore these waiters will
    automatically retry on common (temporary) AWS failures.

    This class should be treated as an abstract class and subclassed before use.
    A subclass should:
    - create the necessary client to pass to BaseWaiterFactory.__init__
    - override _BaseWaiterFactory._waiter_model_data to return the data defining
      the waiter

    Usage:
    waiter_factory = BaseWaiterFactory(module, client)
    waiter = waiters.get_waiter('my_waiter_name')
    waiter.wait(**params)
    """
    module = None
    client = None

    def __init__(self, module, client):
        self.module = module
        self.client = client
        data = self._inject_ratelimit_retries(self._waiter_model_data)
        self._model = botocore.waiter.WaiterModel(waiter_config=dict(version=2, waiters=data))

    @property
    def _waiter_model_data(self):
        """
        Subclasses should override this method to return a dictionary mapping
        waiter names to the waiter definition.

        This data is similar to the data found in botocore's waiters-2.json
        files (for example: botocore/botocore/data/ec2/2016-11-15/waiters-2.json)
        with two differences:
        1) Waiter names do not have transformations applied during lookup
        2) Only the 'waiters' data is required, the data is assumed to be
           version 2

        for example:

        @property
        def _waiter_model_data(self):
            return dict(
                tgw_attachment_deleted=dict(
                    operation='DescribeTransitGatewayAttachments',
                    delay=5, maxAttempts=120,
                    acceptors=[
                        dict(state='retry', matcher='pathAll', expected='deleting', argument='TransitGatewayAttachments[].State'),
                        dict(state='success', matcher='pathAll', expected='deleted', argument='TransitGatewayAttachments[].State'),
                        dict(state='success', matcher='path', expected=True, argument='length(TransitGatewayAttachments[]) == `0`'),
                        dict(state='success', matcher='error', expected='InvalidRouteTableID.NotFound'),
                    ]
                ),
            )

        or

        @property
        def _waiter_model_data(self):
            return {
                "instance_exists": {
                    "delay": 5,
                    "maxAttempts": 40,
                    "operation": "DescribeInstances",
                    "acceptors": [
                        {
                            "matcher": "path",
                            "expected": true,
                            "argument": "length(Reservations[]) > `0`",
                            "state": "success"
                        },
                        {
                            "matcher": "error",
                            "expected": "InvalidInstanceID.NotFound",
                            "state": "retry"
                        }
                    ]
                },
            }
        """
        return dict()

    def _inject_ratelimit_retries(self, model):
        extra_retries = ['RequestLimitExceeded', 'Unavailable', 'ServiceUnavailable', 'InternalFailure', 'InternalError', 'TooManyRequestsException', 'Throttling']
        acceptors = []
        for error in extra_retries:
            acceptors.append(dict(state='retry', matcher='error', expected=error))
        _model = deepcopy(model)
        for waiter in _model:
            _model[waiter]['acceptors'].extend(acceptors)
        return _model

    def get_waiter(self, waiter_name):
        waiters = self._model.waiter_names
        if waiter_name not in waiters:
            self.module.fail_json(f'Unable to find waiter {waiter_name}.  Available_waiters: {waiters}')
        return botocore.waiter.create_waiter_with_client(waiter_name, self._model, self.client)