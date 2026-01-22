from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
class Boto3Mixin:

    @staticmethod
    def aws_error_handler(description):
        """
        A simple wrapper that handles the usual botocore exceptions and exits
        with module.fail_json_aws.  Designed to be used with BaseResourceManager.
        Assumptions:
          1) First argument (usually `self` of method being wrapped will have a
             'module' attribute which is an AnsibleAWSModule
          2) First argument of method being wrapped will have an
            _extra_error_output() method which takes no arguments and returns a
            dictionary of extra parameters to be returned in the event of a
            botocore exception.
        Parameters:
          description (string): In the event of a botocore exception the error
                                message will be 'Failed to {DESCRIPTION}'.

        Example Usage:
            class ExampleClass(Boto3Mixin):
                def __init__(self, module)
                    self.module = module
                    self._get_client()

                @Boto3Mixin.aws_error_handler("connect to AWS")
                def _get_client(self):
                    self.client = self.module.client('ec2')

                @Boto3Mixin.aws_error_handler("describe EC2 instances")
                def _do_something(**params):
                    return self.client.describe_instances(**params)
        """

        def wrapper(func):

            @wraps(func)
            def handler(_self, *args, **kwargs):
                extra_ouput = _self._extra_error_output()
                try:
                    return func(_self, *args, **kwargs)
                except botocore.exceptions.WaiterError as e:
                    _self.module.fail_json_aws(e, msg=f'Failed waiting for {description}', **extra_ouput)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    _self.module.fail_json_aws(e, msg=f'Failed to {description}', **extra_ouput)
            return handler
        return wrapper

    def _normalize_boto3_resource(self, resource, add_tags=False):
        """
        Performs common boto3 resource to Ansible resource conversion.
        `resource['Tags']` will by default be converted from the boto3 tag list
        format to a simple dictionary.
        Parameters:
          resource (dict): The boto3 style resource to convert to the normal Ansible
                           format (snake_case).
          add_tags (bool): When `true`, if a resource does not have 'Tags' property
                           the returned resource will have tags set to an empty
                           dictionary.
        """
        if resource is None:
            return None
        tags = resource.get('Tags', None)
        if tags:
            tags = boto3_tag_list_to_ansible_dict(tags)
        elif add_tags or tags is not None:
            tags = {}
        normalized_resource = camel_dict_to_snake_dict(resource)
        if tags is not None:
            normalized_resource['tags'] = tags
        return normalized_resource

    def _extra_error_output(self):
        return dict()