from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class ApitoolsClientDef(object):
    """Struct for info required to instantiate clients/messages for API versions.

  Attributes:
    class_path: str, Path to the package containing api related modules.
    client_classpath: str, Relative path to the client class for an API version.
    client_full_classpath: str, Full path to the client class for an API
      version.
    messages_modulepath: str, Relative path to the messages module for an API
      version.
    messages_full_modulepath: str, Full path to the messages module for an API
      version.
    base_url: str, The base_url used for the default version of the API.
  """

    def __init__(self, class_path, client_classpath, messages_modulepath, base_url):
        self.class_path = class_path
        self.client_classpath = client_classpath
        self.messages_modulepath = messages_modulepath
        self.base_url = base_url

    @property
    def client_full_classpath(self):
        return self.class_path + '.' + self.client_classpath

    @property
    def messages_full_modulepath(self):
        return self.class_path + '.' + self.messages_modulepath

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_init_source(self):
        src_fmt = 'ApitoolsClientDef("{0}", "{1}", "{2}", "{3}")'
        return src_fmt.format(self.class_path, self.client_classpath, self.messages_modulepath, self.base_url)

    def __repr__(self):
        return self.get_init_source()