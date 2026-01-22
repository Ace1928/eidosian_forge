import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
class TFSModelEndpoint(object):

    def __init__(self, url: str, config: TFSModelConfig, ver: str='v1', sess: requests.Session=requests.Session(), headers: Dict[str, Any]={}):
        self.url = url
        self.config = config
        self.ver = ver
        self.sess = sess
        self.headers = headers
        self.validate_endpoints()

    @property
    def default_endpoint(self):
        return f'{self.url}/{self.ver}/models/{self.config.name}'

    @property
    def default_url(self):
        if self.config.default_label:
            return self.default_endpoint + f'/labels/{self.config.default_label}'
        if self.config.default_version:
            return self.default_endpoint + f'/versions/{self.config.default_version}'
        return self.default_endpoint

    @property
    def default_predict(self):
        return self.default_url + ':predict'

    def get_label(self, label: str='latest'):
        return self.default_endpoint + '/labels/' + label

    def get_version(self, ver: Union[str, int]):
        return self.default_endpoint + '/versions/' + str(ver)

    def get_endpoint(self, label: str=None, ver: Union[str, int]=None):
        if label:
            return self.get_label(label)
        if ver:
            return self.get_version(ver)
        return self.default_url

    def get_predict_endpoint(self, label: str=None, ver: Union[str, int]=None):
        if label:
            return self.get_label(label) + ':predict'
        if ver:
            return self.get_version(ver) + ':predict'
        return self.default_predict

    @timed_cache(seconds=600)
    def get_metadata(self, label: str=None, ver: Union[str, int]=None):
        ep = self.get_endpoint(label=label, ver=ver)
        return self.sess.get(ep + '/metadata', verify=False).json()

    async def get_aio_metadata(self, label: str=None, ver: Union[str, int]=None, **kwargs):
        epoint = self.get_endpoint(label=label, ver=ver)
        async with aiohttp.ClientSession(headers=self.headers) as sess:
            async with sess.get(epoint + '/metadata', verify_ssl=False, **kwargs) as resp:
                return await resp.json()

    async def aio_predict(self, data, label: str=None, ver: Union[str, int]=None, **kwargs):
        epoint = self.get_predict_endpoint(label=label, ver=ver)
        item = TFSRequest(data=data)
        async with aiohttp.ClientSession(headers=self.headers) as sess:
            async with sess.post(epoint, json=item.to_data(), verify_ssl=False, **kwargs) as resp:
                return await resp.json()

    @timed_cache(seconds=60)
    def predict(self, data, label: str=None, ver: Union[str, int]=None, **kwargs):
        epoint = self.get_predict_endpoint(label=label, ver=ver)
        item = TFSRequest(data=data)
        res = self.sess.post(epoint, json=item.to_data(), verify=False)
        return res.json()

    @property
    def is_alive(self):
        return bool(self.get_metadata().get('model_spec'))

    def validate_endpoints(self):
        for n, version in enumerate(self.config.model_versions):
            r = self.get_metadata(label=version.label)
            if r.get('error'):
                self.config.model_versions[n].label = None
            r = self.get_metadata(ver=str(version.step))
            if r.get('error'):
                self.config.model_versions[n].step = None