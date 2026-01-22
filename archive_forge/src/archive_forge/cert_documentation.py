from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Get a cert for a particular instance, given its common name.

  In the SQL API, the last parameter of the URL is the sha1fingerprint, which is
  not something writeable or readable by humans. Instead, the CLI will ask for
  the common name. To allow this, we first query all the ssl certs for the
  instance, and iterate through them to find the one with the correct common
  name.

  Args:
    sql_client: apitools.BaseApiClient, A working client for the sql version to
        be used.
    sql_messages: module, The module that defines the messages for the sql
        version to be used.
    instance_ref: resources.Resource, The instance whos ssl cert is being
        fetched.
    common_name: str, The common name of the ssl cert to be fetched.

  Returns:
    resources.Resource, A ref for the ssl cert being fetched. Or None if it
    could not be found.
  