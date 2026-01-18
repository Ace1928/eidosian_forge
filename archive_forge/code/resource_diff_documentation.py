from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import difflib
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
Prints the unified diff of formatter output for original and changed.

    Prints a unified diff, eg,
    ---

    +++

    @@ -27,6 +27,6 @@

     settings.pricingPlan:                             PER_USE
     settings.replicationType:                         SYNCHRONOUS
     settings.settingsVersion:                         1
    -settings.tier:                                    D1
    +settings.tier:                                    D0
     state:                                            RUNNABLE

    Args:
      print_format: The print format name.
      out: The output stream, stdout if None.
      defaults: Optional resource_projection_spec.ProjectionSpec defaults.
    