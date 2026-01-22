from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def IsJsonOperationQuotaError(error):
    """Returns true if the given loaded json is an operation quota exceeded error.
  """
    try:
        for item in error.get('details'):
            try:
                if item.get('reason') == 'CONCURRENT_OPERATIONS_QUOTA_EXCEEDED':
                    return True
            except (KeyError, AttributeError, TypeError):
                pass
    except (KeyError, AttributeError, TypeError):
        return False
    return False