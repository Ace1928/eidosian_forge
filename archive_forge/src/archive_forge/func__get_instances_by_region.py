import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_instances_by_region(self, regions, filters, strict_permissions):
    """
        :param regions: a list of regions in which to describe instances
        :param filters: a list of boto3 filter dictionaries
        :param strict_permissions: a boolean determining whether to fail or ignore 403 error codes
        :return A list of instance dictionaries
        """
    all_instances = []
    if not any((f['Name'] == 'instance-state-name' for f in filters)):
        filters.append({'Name': 'instance-state-name', 'Values': ['running', 'pending', 'stopping', 'stopped']})
    for connection, _region in self.all_clients('ec2'):
        try:
            reservations = _describe_ec2_instances(connection, filters).get('Reservations')
            instances = []
            for r in reservations:
                new_instances = r['Instances']
                reservation_details = {'OwnerId': r['OwnerId'], 'RequesterId': r.get('RequesterId', ''), 'ReservationId': r['ReservationId']}
                for instance in new_instances:
                    instance.update(reservation_details)
                instances.extend(new_instances)
        except is_boto3_error_code('UnauthorizedOperation') as e:
            if not strict_permissions:
                continue
            self.fail_aws('Failed to describe instances', exception=e)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.fail_aws('Failed to describe instances', exception=e)
        all_instances.extend(instances)
    return all_instances