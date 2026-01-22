import asyncio
import json
import os
from tempfile import TemporaryDirectory
import pytest
import tornado
@pytest.mark.slow
class TestBuildAPI:

    def tempdir(self):
        td = TemporaryDirectory()
        self.tempdirs.append(td)
        return td.name

    def setUp(self):
        self.tempdirs = []

        @self.addCleanup
        def cleanup_tempdirs():
            for d in self.tempdirs:
                d.cleanup()

    async def test_get_status(self, build_api_tester):
        """Make sure there are no kernels running at the start"""
        r = await build_api_tester.getStatus()
        res = r.body.decode()
        resp = json.loads(res)
        assert 'status' in resp
        assert 'message' in resp

    @pytest.mark.skipif(os.name == 'nt', reason='Currently failing on windows')
    async def test_build(self, build_api_tester):
        r = await build_api_tester.build()
        assert r.code == 200

    @pytest.mark.skipif(os.name == 'nt', reason='Currently failing on windows')
    async def test_clear(self, build_api_tester):
        with pytest.raises(tornado.httpclient.HTTPClientError) as e:
            r = await build_api_tester.clear()
            res = r.body.decode()
        assert expected_http_error(e, 500)
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(build_api_tester.build(), loop=loop)
        while True:
            r = await build_api_tester.getStatus()
            res = r.body.decode()
            resp = json.loads(res)
            if resp['status'] == 'building':
                break
        r = await build_api_tester.clear()
        assert r.code == 204