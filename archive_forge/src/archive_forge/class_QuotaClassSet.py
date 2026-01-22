from cinderclient import base
class QuotaClassSet(base.Resource):

    @property
    def id(self):
        """Needed by base.Resource to self-refresh and be indexed."""
        return self.class_name

    def update(self, *args, **kwargs):
        return self.manager.update(self.class_name, *args, **kwargs)