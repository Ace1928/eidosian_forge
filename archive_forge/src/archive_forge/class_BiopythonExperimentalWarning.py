import os
import warnings
class BiopythonExperimentalWarning(BiopythonWarning):
    """Biopython experimental code warning.

    Biopython uses this warning for experimental code ('alpha' or 'beta'
    level code) which is released as part of the standard releases to mark
    sub-modules or functions for early adopters to test & give feedback.

    Code issuing this warning is likely to change (or even be removed) in
    a subsequent release of Biopython. Such code should NOT be used for
    production/stable code. It should only be used if:

    - You are running the latest release of Biopython, or ideally the
      latest code from our repository.
    - You are subscribed to the biopython-dev mailing list to provide
      feedback on this code, and to be alerted of changes to it.

    If all goes well, experimental code would be promoted to stable in
    a subsequent release, and this warning removed from it.
    """