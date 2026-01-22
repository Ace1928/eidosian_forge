import json
import logging
from typing import Dict, NamedTuple, Optional, Union
import urllib
from absl import flags
from utils import bq_consts
from utils import bq_error
def _get_tpc_service_endpoint_hostname(service: Service=Service.BIGQUERY, universe_domain: str='googleapis.com', region: Optional[str]=None, is_mtls: bool=False, is_rep: bool=False, is_lep: bool=False) -> str:
    """Returns the TPC service endpoint hostname."""
    logging.info('Building a root URL for the %s service in the "%s" universe for region "%s" %s mTLS, %s REP, and %s LEP', service, universe_domain, region, 'with' if is_mtls else 'without', 'with' if is_rep else 'without', 'with' if is_lep else 'without')
    if is_mtls and is_rep and region:
        return f'{service}.{region}.rep.mtls.{universe_domain}'
    if not is_mtls and is_rep and region:
        return f'{service}.{region}.rep.{universe_domain}'
    if is_mtls and (not region):
        return f'{service}.mtls.{universe_domain}'
    if not is_mtls and is_lep and region:
        return f'{region}-{service}.{universe_domain}'
    return f'{service}.{universe_domain}'