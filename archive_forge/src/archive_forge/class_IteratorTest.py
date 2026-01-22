import cherrypy
from cherrypy.test import helper
class IteratorTest(helper.CPWebCase):

    @staticmethod
    def setup_server():

        class Root(object):

            @cherrypy.expose
            def count(self, clsname):
                cherrypy.response.headers['Content-Type'] = 'text/plain'
                return str(globals()[clsname].created)

            @cherrypy.expose
            def getall(self, clsname):
                cherrypy.response.headers['Content-Type'] = 'text/plain'
                return globals()[clsname]()

            @cherrypy.expose
            @cherrypy.config(**{'response.stream': True})
            def stream(self, clsname):
                return self.getall(clsname)
        cherrypy.tree.mount(Root())

    def test_iterator(self):
        try:
            self._test_iterator()
        except Exception:
            'Test fails intermittently. See #1419'

    def _test_iterator(self):
        if cherrypy.server.protocol_version != 'HTTP/1.1':
            return self.skip()
        self.PROTOCOL = 'HTTP/1.1'
        closables = ['OurClosableIterator', 'OurGenerator']
        unclosables = ['OurUnclosableIterator', 'OurNotClosableIterator']
        all_classes = closables + unclosables
        import random
        random.shuffle(all_classes)
        for clsname in all_classes:
            self.getPage('/count/' + clsname)
            self.assertStatus(200)
            self.assertBody('0')
        for clsname in all_classes:
            itr_conn = self.get_conn()
            itr_conn.putrequest('GET', '/getall/' + clsname)
            itr_conn.endheaders()
            response = itr_conn.getresponse()
            self.assertEqual(response.status, 200)
            headers = response.getheaders()
            for header_name, header_value in headers:
                if header_name.lower() == 'content-length':
                    expected = str(1024 * 16 * 256)
                    assert header_value == expected, header_value
                    break
            else:
                raise AssertionError('No Content-Length header found')
            self.getPage('/count/' + clsname)
            self.assertStatus(200)
            self.assertBody('0')
            itr_conn.close()
        stream_counts = {}
        for clsname in all_classes:
            itr_conn = self.get_conn()
            itr_conn.putrequest('GET', '/stream/' + clsname)
            itr_conn.endheaders()
            response = itr_conn.getresponse()
            self.assertEqual(response.status, 200)
            response.fp.read(65536)
            self.getPage('/count/' + clsname)
            self.assertBody('1')
            itr_conn.close()
            self.getPage('/count/' + clsname)
            if clsname in closables:
                if self.body != '0':
                    import time
                    time.sleep(0.1)
                    self.getPage('/count/' + clsname)
            stream_counts[clsname] = int(self.body)
        for clsname in closables:
            assert stream_counts[clsname] == 0, 'did not close off stream response correctly, expected count of zero for %s: %s' % (clsname, stream_counts)