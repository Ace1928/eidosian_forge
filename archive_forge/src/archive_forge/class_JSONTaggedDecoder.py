import json
class JSONTaggedDecoder(json.JSONDecoder):

    def decode(self, s):
        return self.decode_obj(super().decode(s))

    @classmethod
    def decode_obj(cls, obj):
        if isinstance(obj, dict):
            obj = {key: cls.decode_obj(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            obj = list((cls.decode_obj(val) for val in obj))
        if not isinstance(obj, dict) or len(obj) != 1:
            return obj
        obj_tag = next(iter(obj.keys()))
        if not obj_tag.startswith('!'):
            return obj
        if obj_tag not in json_tags:
            raise ValueError('Unknown tag', obj_tag)
        obj_cls = json_tags[obj_tag]
        return obj_cls.decode_json_obj(obj[obj_tag])