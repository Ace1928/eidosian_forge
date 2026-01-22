import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def _inject_limit_retries(model):
    extra_retries = ['RequestLimitExceeded', 'Unavailable', 'ServiceUnavailable', 'InternalFailure', 'InternalError', 'TooManyRequestsException', 'Throttling']
    acceptors = []
    for error in extra_retries:
        acceptors.append({'state': 'success', 'matcher': 'error', 'expected': error})
    _model = copy.deepcopy(model)
    for waiter in model['waiters']:
        _model['waiters'][waiter]['acceptors'].extend(acceptors)
    return _model