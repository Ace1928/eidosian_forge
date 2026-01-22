from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.resource import yaml_printer
class ExportPrinter(yaml_printer.YamlPrinter):
    """Printer for k8s_objects to export.

  Omits status information, and metadata that isn't consistent across
  deployments, like project or region.
  """

    def _AddRecord(self, record, delimit=True):
        record = self._FilterForExport(record)
        super(ExportPrinter, self)._AddRecord(record, delimit)

    def _FilterForExport(self, record):
        m = copy.deepcopy(record)
        meta = m.get('metadata')
        if meta:
            meta.pop('creationTimestamp', None)
            meta.pop('generation', None)
            meta.pop('resourceVersion', None)
            meta.pop('selfLink', None)
            meta.pop('uid', None)
            for k in _OMITTED_ANNOTATIONS:
                meta.get('annotations', {}).pop(k, None)
        m.pop('status', None)
        return m