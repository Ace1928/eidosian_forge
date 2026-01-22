from __future__ import annotations
import logging
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from pymatgen.core import DummySpecies, Structure
from pymatgen.util.due import Doi, due
from pymatgen.util.provenance import StructureNL
def _parse_provider_link(provider, provider_link_json):
    """No validation attempted."""
    ps = {}
    try:
        data = [dct for dct in provider_link_json['data'] if dct['attributes']['link_type'] == 'child']
        for link in data:
            key = f'{provider}.{link['id']}' if provider != link['id'] else provider
            if link['attributes']['base_url']:
                ps[key] = Provider(name=link['attributes']['name'], base_url=link['attributes']['base_url'], description=link['attributes']['description'], homepage=link['attributes'].get('homepage'), prefix=link['attributes'].get('prefix'))
    except Exception:
        pass
    return ps