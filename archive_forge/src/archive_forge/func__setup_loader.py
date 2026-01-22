import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
def _setup_loader(self):
    """
        Setup loader paths so that we can load resources.
        """
    self._loader = self._session.get_component('data_loader')
    self._loader.search_paths.append(os.path.join(os.path.dirname(__file__), 'data'))