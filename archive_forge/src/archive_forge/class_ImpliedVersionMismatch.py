import os_service_types
from keystoneauth1.exceptions import base
class ImpliedVersionMismatch(ValueError):
    label = 'version'

    def __init__(self, service_type, implied, given):
        super(ImpliedVersionMismatch, self).__init__('service_type {service_type} was given which implies major API version {implied} but {label} of {given} was also given. Please update your code to use the official service_type {official_type}.'.format(service_type=service_type, implied=str(implied[0]), given=given, label=self.label, official_type=_SERVICE_TYPES.get_service_type(service_type)))