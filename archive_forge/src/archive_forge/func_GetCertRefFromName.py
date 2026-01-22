from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetCertRefFromName(sql_client, sql_messages, resources, instance_ref, common_name):
    """Get a cert reference for a particular instance, given its common name.

  Args:
    sql_client: apitools.BaseApiClient, A working client for the sql version to
        be used.
    sql_messages: module, The module that defines the messages for the sql
        version to be used.
    resources: resources.Registry, The registry that can create resource refs
        for the sql version to be used.
    instance_ref: resources.Resource, The instance whos ssl cert is being
        fetched.
    common_name: str, The common name of the ssl cert to be fetched.

  Returns:
    resources.Resource, A ref for the ssl cert being fetched. Or None if it
    could not be found.
  """
    cert = GetCertFromName(sql_client, sql_messages, instance_ref, common_name)
    if not cert:
        return None
    return resources.Create(collection='sql.sslCerts', project=instance_ref.project, instance=instance_ref.instance, sha1Fingerprint=cert.sha1Fingerprint)