import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
@dataclass(config=dataclass_config)
class Gallery(Block):
    items: LList[Union[GalleryReport, GalleryURL]] = Field(default_factory=list)

    def to_model(self):
        links = []
        for x in self.items:
            if isinstance(x, GalleryReport):
                link = internal.GalleryLinkReport(id=x.report_id)
            elif isinstance(x, GalleryURL):
                link = internal.GalleryLinkURL(url=x.url, title=x.title, description=x.description, image_url=x.image_url)
            links.append(link)
        return internal.Gallery(links=links)

    @classmethod
    def from_model(cls, model: internal.Gallery):
        items = []
        if model.ids:
            items = [GalleryReport(x) for x in model.ids]
        elif model.links:
            for x in model.links:
                if isinstance(x, internal.GalleryLinkReport):
                    item = GalleryReport(report_id=x.id)
                elif isinstance(x, internal.GalleryLinkURL):
                    item = GalleryURL(url=x.url, title=x.title, description=x.description, image_url=x.image_url)
                items.append(item)
        return cls(items=items)