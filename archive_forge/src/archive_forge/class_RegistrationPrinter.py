from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
class RegistrationPrinter(cp.CustomPrinterBase):
    """Prints the Cloud Domains registration in YAML format with custom fields order."""
    _KNOWN_FIELDS_BY_IMPORTANCE = ['name', 'createTime', 'domainName', 'state', 'issues', 'expireTime', 'labels', 'managementSettings', 'dnsSettings', 'contactSettings', 'pendingContactSettings', 'supportedPrivacy']
    _KNOWN_REPEATED_FIELDS = ['issues', 'supportedPrivacy']

    def _ClearField(self, registration, field):
        if field in self._KNOWN_REPEATED_FIELDS:
            setattr(registration, field, [])
        else:
            setattr(registration, field, None)

    def _TransformKnownFields(self, printer, registration):
        for field in self._KNOWN_FIELDS_BY_IMPORTANCE:
            record = getattr(registration, field, None)
            if record:
                printer.AddRecord({field: record}, delimit=False)

    def _TransformRemainingFields(self, printer, registration):
        for field in self._KNOWN_FIELDS_BY_IMPORTANCE:
            if getattr(registration, field, None):
                self._ClearField(registration, field)
        finished = True
        if registration.all_unrecognized_fields():
            finished = False
        for f in registration.all_fields():
            if getattr(registration, f.name):
                finished = False
        if not finished:
            printer.AddRecord(registration, delimit=False)

    def Transform(self, registration):
        """Transform a registration into a YAML output."""
        yaml = yp.YamlPrinter()
        self._TransformKnownFields(yaml, registration)
        self._TransformRemainingFields(yaml, registration)