import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@classmethod
def _get_gcp_context(cls):
    ctx = {'cloud.provider': CLOUD_PROVIDER.GCP, 'cloud.platform': CLOUD_PLATFORM.GCP_COMPUTE_ENGINE}
    try:
        if cls.gcp_metadata is None:
            r = cls.http.request('GET', GCP_METADATA_URL, headers={'Metadata-Flavor': 'Google'})
            if r.status != 200:
                return ctx
            cls.gcp_metadata = json.loads(r.data.decode('utf-8'))
        try:
            ctx['cloud.account.id'] = cls.gcp_metadata['project']['projectId']
        except Exception:
            pass
        try:
            ctx['cloud.availability_zone'] = cls.gcp_metadata['instance']['zone'].split('/')[-1]
        except Exception:
            pass
        try:
            ctx['cloud.region'] = cls.gcp_metadata['instance']['region'].split('/')[-1]
        except Exception:
            pass
        try:
            ctx['host.id'] = cls.gcp_metadata['instance']['id']
        except Exception:
            pass
    except Exception:
        pass
    return ctx