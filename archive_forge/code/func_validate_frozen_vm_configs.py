import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def validate_frozen_vm_configs(conf: dict):
    """
    valid frozen VM configs are:
    1. ``ray up`` on a frozen VM to be deployed from an OVF template:
    frozen_vm:
        name: single-frozen-vm
        library_item: frozen-vm-template
        cluster: vsanCluster
        datastore: vsanDatastore

    2. ``ray up`` on an existing frozen VM:
        frozen_vm:
            name: existing-single-frozen-vm

    3. ``ray up`` on a resource pool of frozen VMs to be deployed from an OVF template:
        frozen_vm:
            name: frozen-vm-prefix
            library_item: frozen-vm-template
            resource_pool: frozen-vm-resource-pool
            datastore: vsanDatastore

    4. ``ray up`` on an existing resource pool of frozen VMs:
        frozen_vm:
            resource_pool: frozen-vm-resource-pool
    This function will throw an Exception if the config doesn't lie in above examples
    """
    if conf.get('library_item'):
        if not conf.get('datastore'):
            raise ValueError("'datastore' is not given when trying to deploy the frozen VM from OVF.")
        if not (conf.get('cluster') or conf.get('resource_pool')):
            raise ValueError("both 'cluster' and 'resource_pool' are missing when trying to deploy the frozen VM from OVF, at least one should be given.")
        if not conf.get('name'):
            raise ValueError("'name' must be given when deploying the frozen VM from OVF.")
    elif not ('name' in conf or 'resource_pool' in conf):
        raise ValueError("both 'name' and 'resource_pool' are missing, at least one should be given for the frozen VM(s).")