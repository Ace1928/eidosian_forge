import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
class TestZenodoAPISupport:
    """
    Test support for different Zenodo APIs
    """
    article_id = 123456
    doi = f'10.0001/zenodo.{article_id}'
    doi_url = f'https://doi.org/{doi}'
    file_name = 'my-file.zip'
    file_url = 'https://zenodo.org/api/files/513d7033-93a2-4eeb-821c-2fb0bbab0012/my-file.zip'
    file_checksum = '2942bfabb3d05332b66eb128e0842cff'
    legacy_api_response = {'created': '2021-20-19T08:00:00.000000+00:00', 'modified': '2021-20-19T08:00:00.000000+00:00', 'id': article_id, 'doi': doi, 'doi_url': doi_url, 'files': [{'id': '513d7033-93a2-4eeb-821c-2fb0bbab0012', 'key': file_name, 'checksum': f'md5:{file_checksum}', 'links': {'self': file_url}}]}
    new_api_response = {'created': '2021-20-19T08:00:00.000000+00:00', 'modified': '2021-20-19T08:00:00.000000+00:00', 'id': article_id, 'doi': doi, 'doi_url': doi_url, 'files': [{'id': '513d7033-93a2-4eeb-821c-2fb0bbab0012', 'filename': file_name, 'checksum': file_checksum, 'links': {'self': file_url}}]}
    invalid_api_response = {'created': '2021-20-19T08:00:00.000000+00:00', 'modified': '2021-20-19T08:00:00.000000+00:00', 'id': article_id, 'doi': doi, 'doi_url': doi_url, 'files': [{'id': '513d7033-93a2-4eeb-821c-2fb0bbab0012', 'filename': file_name, 'checksum': file_checksum, 'links': {'self': file_url}}, {'id': '513d7033-93a2-4eeb-821c-2fb0bbab0012', 'key': file_name, 'checksum': f'md5:{file_checksum}', 'links': {'self': file_url}}]}

    @pytest.mark.parametrize('api_version, api_response', [('legacy', legacy_api_response), ('new', new_api_response), ('invalid', invalid_api_response)])
    def test_api_version(self, httpserver, api_version, api_response):
        """
        Test if the API version is correctly detected.
        """
        httpserver.expect_request(f'/zenodo.{self.article_id}').respond_with_json(api_response)
        downloader = ZenodoRepository(doi=self.doi, archive_url=self.doi_url)
        downloader.base_api_url = httpserver.url_for('')
        if api_version != 'invalid':
            assert downloader.api_version == api_version
        else:
            msg = "Couldn't determine the version of the Zenodo API"
            with pytest.raises(ValueError, match=msg):
                api_version = downloader.api_version

    @pytest.mark.parametrize('api_version, api_response', [('legacy', legacy_api_response), ('new', new_api_response)])
    def test_download_url(self, httpserver, api_version, api_response):
        """
        Test if the download url is correct for each API version.
        """
        httpserver.expect_request(f'/zenodo.{self.article_id}').respond_with_json(api_response)
        downloader = ZenodoRepository(doi=self.doi, archive_url=self.doi_url)
        downloader.base_api_url = httpserver.url_for('')
        download_url = downloader.download_url(file_name=self.file_name)
        if api_version == 'legacy':
            assert download_url == self.file_url
        else:
            expected_url = f'https://zenodo.org/records/{self.article_id}/files/{self.file_name}?download=1'
            assert download_url == expected_url

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