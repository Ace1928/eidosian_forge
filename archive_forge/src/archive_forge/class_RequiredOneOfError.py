from __future__ import absolute_import, division, print_function
class RequiredOneOfError(AnsibleValidationError):
    """Error with parameters where at least one is required"""