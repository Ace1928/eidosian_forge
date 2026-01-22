from __future__ import annotations
import json
import re
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_layer_properties(self, lyr_desc: Optional[str]=None) -> dict:
    """Get the layer properties from the FeatureLayer."""
    import arcgis
    layer_number_pattern = re.compile('/\\d+$')
    props = self.layer.properties
    if lyr_desc is None:
        try:
            if self.BEAUTIFULSOUP:
                lyr_desc = self.BEAUTIFULSOUP(props['description']).text
            else:
                lyr_desc = props['description']
            lyr_desc = lyr_desc or _NOT_PROVIDED
        except KeyError:
            lyr_desc = _NOT_PROVIDED
    try:
        item_id = props['serviceItemId']
        item = self.gis.content.get(item_id) or arcgis.features.FeatureLayer(re.sub(layer_number_pattern, '', self.url))
        try:
            raw_desc = item.description
        except AttributeError:
            raw_desc = item.properties.description
        if self.BEAUTIFULSOUP:
            item_desc = self.BEAUTIFULSOUP(raw_desc).text
        else:
            item_desc = raw_desc
        item_desc = item_desc or _NOT_PROVIDED
    except KeyError:
        item_desc = _NOT_PROVIDED
    return {'layer_description': lyr_desc, 'item_description': item_desc, 'layer_properties': props}