import traceback
from fastapi.exceptions import HTTPException
from lazyops.utils.logs import logger
from typing import Optional
class NoResultsFoundException(ORMException):
    """
    No Results Found Exception
    """
    base = 'No Results Found'
    default_status_code = 404