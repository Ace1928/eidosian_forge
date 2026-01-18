import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.parametrize('api_response', [legacy_api_response, new_api_response])
def test_populate_registry(self, httpserver, tmp_path, api_response):
    """
        Test if population of registry is correctly done for each API version.
        """
    httpserver.expect_request(f'/zenodo.{self.article_id}').respond_with_json(api_response)
    puppy = Pooch(base_url='', path=tmp_path)
    downloader = ZenodoRepository(doi=self.doi, archive_url=self.doi_url)
    downloader.base_api_url = httpserver.url_for('')
    downloader.populate_registry(puppy)
    assert puppy.registry == {self.file_name: f'md5:{self.file_checksum}'}