from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_property
def _ConfigSections(self, record):
    config = record.get('spec', {}).get('config', {})
    sections = []
    for section_name, data in sorted(config.items()):
        title = _ConfigTitle(section_name)
        section = cp.Section([cp.Labeled([(title, _ConfigSectionData(data))])])
        sections.append(section)
    return sections