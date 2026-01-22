import copy
import json
import jsonpatch
import warlock.model as warlock
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