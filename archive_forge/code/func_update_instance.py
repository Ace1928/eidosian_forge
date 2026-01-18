import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def update_instance(self, instance_id, layer_ids=None, instance_type=None, auto_scaling_type=None, hostname=None, os=None, ami_id=None, ssh_key_name=None, architecture=None, install_updates_on_boot=None, ebs_optimized=None):
    """
        Updates a specified instance.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID.

        :type layer_ids: list
        :param layer_ids: The instance's layer IDs.

        :type instance_type: string
        :param instance_type: The instance type. AWS OpsWorks supports all
            instance types except Cluster Compute, Cluster GPU, and High Memory
            Cluster. For more information, see `Instance Families and Types`_.
            The parameter values that you use to specify the various types are
            in the API Name column of the Available Instance Types table.

        :type auto_scaling_type: string
        :param auto_scaling_type: For load-based or time-based instances, the
            type.

        :type hostname: string
        :param hostname: The instance host name.

        :type os: string
        :param os: The instance's operating system, which must be set to one of
            the following.

        + Standard operating systems: An Amazon Linux version such as `Amazon
              Linux 2014.09`, `Ubuntu 12.04 LTS`, or `Ubuntu 14.04 LTS`.
        + Custom AMIs: `Custom`


        The default option is the current Amazon Linux version, such as `Amazon
            Linux 2014.09`. If you set this parameter to `Custom`, you must use
            the CreateInstance action's AmiId parameter to specify the custom
            AMI that you want to use. For more information on the standard
            operating systems, see `Operating Systems`_For more information on
            how to use custom AMIs with OpsWorks, see `Using Custom AMIs`_.

        :type ami_id: string
        :param ami_id:
        A custom AMI ID to be used to create the instance. The AMI should be
            based on one of the standard AWS OpsWorks AMIs: Amazon Linux,
            Ubuntu 12.04 LTS, or Ubuntu 14.04 LTS. For more information, see
            `Instances`_

        If you specify a custom AMI, you must set `Os` to `Custom`.

        :type ssh_key_name: string
        :param ssh_key_name: The instance SSH key name.

        :type architecture: string
        :param architecture: The instance architecture. Instance types do not
            necessarily support both architectures. For a list of the
            architectures that are supported by the different instance types,
            see `Instance Families and Types`_.

        :type install_updates_on_boot: boolean
        :param install_updates_on_boot:
        Whether to install operating system and package updates when the
            instance boots. The default value is `True`. To control when
            updates are installed, set this value to `False`. You must then
            update your instances manually by using CreateDeployment to run the
            `update_dependencies` stack command or manually running `yum`
            (Amazon Linux) or `apt-get` (Ubuntu) on the instances.


        We strongly recommend using the default value of `True`, to ensure that
            your instances have the latest security updates.

        :type ebs_optimized: boolean
        :param ebs_optimized: Whether this is an Amazon EBS-optimized instance.

        """
    params = {'InstanceId': instance_id}
    if layer_ids is not None:
        params['LayerIds'] = layer_ids
    if instance_type is not None:
        params['InstanceType'] = instance_type
    if auto_scaling_type is not None:
        params['AutoScalingType'] = auto_scaling_type
    if hostname is not None:
        params['Hostname'] = hostname
    if os is not None:
        params['Os'] = os
    if ami_id is not None:
        params['AmiId'] = ami_id
    if ssh_key_name is not None:
        params['SshKeyName'] = ssh_key_name
    if architecture is not None:
        params['Architecture'] = architecture
    if install_updates_on_boot is not None:
        params['InstallUpdatesOnBoot'] = install_updates_on_boot
    if ebs_optimized is not None:
        params['EbsOptimized'] = ebs_optimized
    return self.make_request(action='UpdateInstance', body=json.dumps(params))