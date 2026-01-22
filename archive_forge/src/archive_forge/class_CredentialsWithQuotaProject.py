import abc
import inspect
import six
from google.auth import credentials
class CredentialsWithQuotaProject(credentials.CredentialsWithQuotaProject):
    """Abstract base for credentials supporting ``with_quota_project`` factory"""