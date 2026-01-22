import copy
import json
import jsonpatch
import warlock.model as warlock
class SchemaBasedModel(warlock.Model):
    """Glance specific subclass of the warlock Model.

    This implementation alters the function of the patch property
    to take into account the schema's core properties. With this version
    undefined properties which are core will generated 'replace'
    operations rather than 'add' since this is what the Glance API
    expects.
    """

    def _make_custom_patch(self, new, original):
        if not self.get('tags'):
            tags_patch = []
        else:
            tags_patch = [{'path': '/tags', 'value': self.get('tags'), 'op': 'replace'}]
        patch_string = jsonpatch.make_patch(original, new).to_string()
        patch = json.loads(patch_string)
        if not patch:
            return json.dumps(tags_patch)
        else:
            return json.dumps(patch + tags_patch)

    @warlock.Model.patch.getter
    def patch(self):
        """Return a jsonpatch object representing the delta."""
        original = copy.deepcopy(self.__dict__['__original__'])
        new = dict(self)
        if self.schema:
            for name, prop in self.schema['properties'].items():
                if name not in original and name in new and prop.get('is_base', True):
                    original[name] = None
        original['tags'] = None
        new['tags'] = None
        return self._make_custom_patch(new, original)