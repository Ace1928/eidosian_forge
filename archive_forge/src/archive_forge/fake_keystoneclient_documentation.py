from keystoneauth1 import session
from heat.common import context
A fake FakeKeystoneClient. This can be used during some runtime
scenarios where you want to disable Heat's internal Keystone dependencies
entirely. One example is the TripleO Undercloud installer.

To use this class at runtime set to following heat.conf config setting:

  keystone_backend = heat.engine.clients.os.keystone.fake_keystoneclient  .FakeKeystoneClient

