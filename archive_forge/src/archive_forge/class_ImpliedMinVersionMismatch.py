import os_service_types
from keystoneauth1.exceptions import base
class ImpliedMinVersionMismatch(ImpliedVersionMismatch):
    label = 'min_version'