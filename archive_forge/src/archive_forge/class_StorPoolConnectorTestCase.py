from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import storpool as connector
from os_brick.tests.initiator import test_connector
class StorPoolConnectorTestCase(test_connector.ConnectorTestCase):

    def volumeName(self, vid):
        return volumeNameExt(vid)

    def get_fake_size(self):
        return self.fakeSize

    def execute(self, *cmd, **kwargs):
        if cmd[0] == 'blockdev':
            self.assertEqual(len(cmd), 3)
            self.assertEqual(cmd[1], '--getsize64')
            self.assertEqual(cmd[2], '/dev/storpool/' + self.volumeName(self.fakeProp['volume']))
            return (str(self.get_fake_size()) + '\n', None)
        raise Exception('Unrecognized command passed to ' + type(self).__name__ + '.execute(): ' + str.join(', ', map(lambda s: "'" + s + "'", cmd)))

    def setUp(self):
        super(StorPoolConnectorTestCase, self).setUp()
        self.fakeProp = {'volume': 'sp-vol-1', 'client_id': 1, 'access_mode': 'rw'}
        self.fakeConnection = None
        self.fakeSize = 1024 * 1024 * 1024
        self.connector = connector.StorPoolConnector(None, execute=self.execute)
        self.adb = self.connector._attach

    def test_connect_volume(self):
        self.assertNotIn(self.volumeName(self.fakeProp['volume']), self.adb.attached)
        conn = self.connector.connect_volume(self.fakeProp)
        self.assertIn('type', conn)
        self.assertIn('path', conn)
        self.assertIn(self.volumeName(self.fakeProp['volume']), self.adb.attached)
        self.assertEqual(self.connector.get_search_path(), '/dev/storpool')
        paths = self.connector.get_volume_paths(self.fakeProp)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], '/dev/storpool/' + self.volumeName(self.fakeProp['volume']))
        self.fakeConnection = conn

    def test_disconnect_volume(self):
        if self.fakeConnection is None:
            self.test_connect_volume()
        self.assertIn(self.volumeName(self.fakeProp['volume']), self.adb.attached)
        self.connector.disconnect_volume(self.fakeProp, None)
        self.assertNotIn(self.volumeName(self.fakeProp['volume']), self.adb.attached)

    def test_connect_exceptions(self):
        """Raise exceptions on missing connection information"""
        fake = self.fakeProp
        for key in fake.keys():
            c = dict(fake)
            del c[key]
            self.assertRaises(exception.BrickException, self.connector.connect_volume, c)
            if key != 'access_mode':
                self.assertRaises(exception.BrickException, self.connector.disconnect_volume, c, None)

    def test_extend_volume(self):
        if self.fakeConnection is None:
            self.test_connect_volume()
        self.fakeSize += 1024 * 1024 * 1024
        size_list = [self.fakeSize, self.fakeSize - 1, self.fakeSize - 2]
        vdata = mock.MagicMock(spec=['size'])
        vdata.size = self.fakeSize
        vdata_list = [[vdata]]

        def fake_volume_list(name):
            self.assertEqual(name, self.adb.volumeName(self.fakeProp['volume']))
            return vdata_list.pop()
        api = mock.MagicMock(spec=['volumeList'])
        api.volumeList = mock.MagicMock(spec=['__call__'])
        with mock.patch.object(self.adb, attribute='api', spec=['__call__']) as fake_api, mock.patch.object(self, attribute='get_fake_size', spec=['__call__']) as fake_size, mock.patch('time.sleep') as fake_sleep:
            fake_api.return_value = api
            api.volumeList.side_effect = fake_volume_list
            fake_size.side_effect = size_list.pop
            newSize = self.connector.extend_volume(self.fakeProp)
            self.assertEqual(fake_api.call_count, 1)
            self.assertEqual(api.volumeList.call_count, 1)
            self.assertListEqual(vdata_list, [])
            self.assertEqual(fake_size.call_count, 3)
            self.assertListEqual(size_list, [])
            self.assertEqual(fake_sleep.call_count, 2)
        self.assertEqual(newSize, self.fakeSize)