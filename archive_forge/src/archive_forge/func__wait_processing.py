from __future__ import annotations
import json
import logging
import time
from typing import List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import validator
from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool
def _wait_processing(self, url: str) -> requests.Response:
    for _ in range(10):
        time.sleep(1)
        audio_analysis_result = self._get_edenai(url)
        temp = audio_analysis_result.json()
        if temp['status'] == 'finished':
            if temp['results'][self.providers[0]]['error'] is not None:
                raise Exception(f'EdenAI returned an unexpected response \n                        {temp['results'][self.providers[0]]['error']}')
            else:
                return audio_analysis_result
    raise Exception('Edenai speech to text job id processing Timed out')