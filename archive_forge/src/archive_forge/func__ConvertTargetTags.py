def _ConvertTargetTags(secure_tags):
    template = '  target_secure_tags {{\n    name = "{name}"\n  }}\n'
    records = map(lambda x: template.format(name=x.name), secure_tags)
    return ''.join(records)