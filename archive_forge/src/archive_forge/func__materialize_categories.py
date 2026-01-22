from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def _materialize_categories(self):
    """Materialize actual categorical values."""
    ErrorMessage.catch_bugs_and_request_email(failure_condition=self._parent is None, extra_log="attempted to materialize categories with parent being 'None'")
    categoricals = self._materializer(self._parent, self._column_name)
    self._categories = categoricals.categories
    self._ordered = categoricals.ordered