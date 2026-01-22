import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class MathpixPDFLoader(BasePDFLoader):
    """Load `PDF` files using `Mathpix` service."""

    def __init__(self, file_path: str, processed_file_format: str='md', max_wait_time_seconds: int=500, should_clean_pdf: bool=False, extra_request_data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
        """Initialize with a file path.

        Args:
            file_path: a file for loading.
            processed_file_format: a format of the processed file. Default is "md".
            max_wait_time_seconds: a maximum time to wait for the response from
             the server. Default is 500.
            should_clean_pdf: a flag to clean the PDF file. Default is False.
            extra_request_data: Additional request data.
            **kwargs: additional keyword arguments.
        """
        self.mathpix_api_key = get_from_dict_or_env(kwargs, 'mathpix_api_key', 'MATHPIX_API_KEY')
        self.mathpix_api_id = get_from_dict_or_env(kwargs, 'mathpix_api_id', 'MATHPIX_API_ID')
        kwargs.pop('mathpix_api_key', None)
        kwargs.pop('mathpix_api_id', None)
        super().__init__(file_path, **kwargs)
        self.processed_file_format = processed_file_format
        self.extra_request_data = extra_request_data if extra_request_data is not None else {}
        self.max_wait_time_seconds = max_wait_time_seconds
        self.should_clean_pdf = should_clean_pdf

    @property
    def _mathpix_headers(self) -> Dict[str, str]:
        return {'app_id': self.mathpix_api_id, 'app_key': self.mathpix_api_key}

    @property
    def url(self) -> str:
        return 'https://api.mathpix.com/v3/pdf'

    @property
    def data(self) -> dict:
        options = {'conversion_formats': {self.processed_file_format: True}, **self.extra_request_data}
        return {'options_json': json.dumps(options)}

    def send_pdf(self) -> str:
        with open(self.file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(self.url, headers=self._mathpix_headers, files=files, data=self.data)
        response_data = response.json()
        if 'error' in response_data:
            raise ValueError(f'Mathpix request failed: {response_data['error']}')
        if 'pdf_id' in response_data:
            pdf_id = response_data['pdf_id']
            return pdf_id
        else:
            raise ValueError('Unable to send PDF to Mathpix.')

    def wait_for_processing(self, pdf_id: str) -> None:
        """Wait for processing to complete.

        Args:
            pdf_id: a PDF id.

        Returns: None
        """
        url = self.url + '/' + pdf_id
        for _ in range(0, self.max_wait_time_seconds, 5):
            response = requests.get(url, headers=self._mathpix_headers)
            response_data = response.json()
            error = response_data.get('error', None)
            error_info = response_data.get('error_info', None)
            if error is not None:
                error_msg = f'Unable to retrieve PDF from Mathpix: {error}'
                if error_info is not None:
                    error_msg += f' ({error_info['id']})'
                raise ValueError(error_msg)
            status = response_data.get('status', None)
            if status == 'completed':
                return
            elif status == 'error':
                raise ValueError('Unable to retrieve PDF from Mathpix')
            else:
                print(f'Status: {status}, waiting for processing to complete')
                time.sleep(5)
        raise TimeoutError

    def get_processed_pdf(self, pdf_id: str) -> str:
        self.wait_for_processing(pdf_id)
        url = f'{self.url}/{pdf_id}.{self.processed_file_format}'
        response = requests.get(url, headers=self._mathpix_headers)
        return response.content.decode('utf-8')

    def clean_pdf(self, contents: str) -> str:
        """Clean the PDF file.

        Args:
            contents: a PDF file contents.

        Returns:

        """
        contents = '\n'.join([line for line in contents.split('\n') if not line.startswith('![]')])
        contents = contents.replace('\\section{', '# ').replace('}', '')
        contents = contents.replace('\\$', '$').replace('\\%', '%').replace('\\(', '(').replace('\\)', ')')
        return contents

    def load(self) -> List[Document]:
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf(pdf_id)
        if self.should_clean_pdf:
            contents = self.clean_pdf(contents)
        metadata = {'source': self.source, 'file_path': self.source, 'pdf_id': pdf_id}
        return [Document(page_content=contents, metadata=metadata)]