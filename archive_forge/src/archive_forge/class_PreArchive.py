from lxml import etree
import urllib
from .search import SearchManager
from .users import Users
from .resources import Project
from .tags import Tags
from .jsonutil import JsonTable
class PreArchive(object):

    def __init__(self, interface):
        self._intf = interface
    '\n    Retrieve the status of a session\n    Parameters\n    ----------\n       triple - A list containing the project, timestamp and session id, in\n       that order.\n    '

    def status(self, triple):
        return JsonTable(self._intf._get_json('/data/prearchive/projects')).where(project=triple[0], timestamp=triple[1], folderName=triple[2]).get('status')
    '\n    Retrieve the contents of the prearchive.\n    '

    def get(self):
        return JsonTable(self._intf._get_json('/data/prearchive/projects'), ['project', 'timestamp', 'folderName']).select(['project', 'timestamp', 'folderName']).as_list()[1:]
    '\n    Retrieve the scans of a give session triple\n    Parameters\n    ----------\n       triple - A list containing the project, timestamp and session id, in\n       that order.\n    '

    def get_scans(self, triple):
        return JsonTable(self._intf._get_json('/data/prearchive/projects/%s/scans' % '/'.join(triple))).get('ID')
    '\n    Retrieve the resource of a session triple\n    Parameters\n    ----------\n       triple - A list containing the project, timestamp and session id, in\n       that order.\n       scan_id - id of the scan\n    '

    def get_resources(self, triple, scan_id):
        return JsonTable(self._intf._get_json('/data/prearchive/projects/%s/scans/%s/resources' % ('/'.join(triple), scan_id))).get('label')
    '\n    Retrieve a list of files in a given session triple\n    Parameters\n    ----------\n       triple - A list containing the project, timestamp and session id, in\n       that order.\n       scan_id - id of the scan\n       resource_id - id of the resource\n    '

    def get_files(self, triple, scan_id, resource_id):
        return JsonTable(self._intf._get_json('/data/prearchive/projects/%s/scans/%s/resources/%s/files' % ('/'.join(triple), scan_id, resource_id))).get('Name')
    '\n    Move multiple sessions to a project in the prearchive asynchronously.\n    If only one session is it is done now.\n\n    This does *not* archive a session.\n\n    Parameters\n    ----------\n       uris - a list of session uris\n       new_project - The name of the project to which to move the sessions.\n    '

    def move(self, uris, new_project):
        add_src = lambda u: urllib.urlencode({'src': u})
        async_ = len(uris) > 1 and 'true' or 'false'
        print(async_)
        post_body = '&'.join(map(add_src, uris) + [urllib.urlencode({'newProject': new_project})] + [urllib.urlencode({'async': async_})])
        request_uri = '/data/services/prearchive/move?format=csv'
        ct = {'content-type': 'application/x-www-form-urlencoded'}
        return self._intf._exec(request_uri, 'POST', post_body, ct)
    '\n    Reinspect the file on the filesystem on the XNAT server and recreate the\n    parameters of the file. Essentially a refresh of the file.\n\n    Be warned that if this session has been scheduled for an operation, that\n    operation is cancelled.\n    Parameters\n    ----------\n       uris - a list of session uris\n       new_project - The name of the project to which to move the sessions.\n    '

    def reset(self, triple):
        post_body = 'action=build'
        request_uri = '/data/prearchive/projects/%s?format=single' % '/'.join(triple)
        ct = {'content-type': 'application/x-www-form-urlencoded'}
        return self._intf._exec(request_uri, 'POST', post_body, ct)
    '\n    Delete  a session from the prearchive\n    Parameters\n    ----------\n       uri - The uri of the session to delete\n    '

    def delete(self, uri):
        post_body = 'src=' + uri + '&' + 'async=false'
        request_uri = '/data/services/prearchive/delete?format=csv'
        ct = {'content-type': 'application/x-www-form-urlencoded'}
        return self._intf._exec(request_uri, 'POST', post_body, ct)
    '\n    Get the uri of the given session.\n    Parameters\n    ----------\n       triple - A list containing the project, timestamp and session id, in\n       that order.\n    '

    def get_uri(self, triple):
        j = JsonTable(self._intf._get_json('/data/prearchive/projects'))
        return j.where(project=triple[0], timestamp=triple[1], folderName=triple[2]).get('url')