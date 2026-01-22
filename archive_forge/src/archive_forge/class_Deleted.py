from boto import handler
import xml.sax
class Deleted(object):
    """
    A successfully deleted object in a multi-object delete request.

    :ivar key: Key name of the object that was deleted.
    
    :ivar version_id: Version id of the object that was deleted.
    
    :ivar delete_marker: If True, indicates the object deleted
        was a DeleteMarker.
        
    :ivar delete_marker_version_id: Version ID of the delete marker
        deleted.
    """

    def __init__(self, key=None, version_id=None, delete_marker=False, delete_marker_version_id=None):
        self.key = key
        self.version_id = version_id
        self.delete_marker = delete_marker
        self.delete_marker_version_id = delete_marker_version_id

    def __repr__(self):
        if self.version_id:
            return '<Deleted: %s.%s>' % (self.key, self.version_id)
        else:
            return '<Deleted: %s>' % self.key

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Key':
            self.key = value
        elif name == 'VersionId':
            self.version_id = value
        elif name == 'DeleteMarker':
            if value.lower() == 'true':
                self.delete_marker = True
        elif name == 'DeleteMarkerVersionId':
            self.delete_marker_version_id = value
        else:
            setattr(self, name, value)