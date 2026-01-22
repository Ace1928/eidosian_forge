from cinderclient import base
class DefaultVolumeType(base.Resource):
    """Default volume types for projects."""

    def __repr__(self):
        return '<DefaultVolumeType: %s>' % self.project_id