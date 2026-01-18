from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_rl_policy(module, blade, local_replica_link):
    """Update Bucket Replica Link"""
    changed = False
    new_cred = local_replica_link.remote.name + '/' + module.params['credential']
    if local_replica_link.paused != module.params['paused'] or local_replica_link.remote_credentials.name != new_cred:
        changed = True
        if not module.check_mode:
            try:
                module.warn('{0}'.format(local_replica_link))
                blade.bucket_replica_links.update_bucket_replica_links(local_bucket_names=[module.params['name']], remote_bucket_names=[local_replica_link.remote_bucket.name], remote_names=[local_replica_link.remote.name], bucket_replica_link=BucketReplicaLink(paused=module.params['paused'], remote_credentials=ObjectStoreRemoteCredentials(name=new_cred)))
            except Exception:
                module.fail_json(msg='Failed to update bucket replica link {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)