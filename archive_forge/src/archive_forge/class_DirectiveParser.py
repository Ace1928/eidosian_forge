import re
class DirectiveParser:
    name = 'directive'

    @staticmethod
    def parse_type(m: re.Match):
        raise NotImplementedError()

    @staticmethod
    def parse_title(m: re.Match):
        raise NotImplementedError()

    @staticmethod
    def parse_content(m: re.Match):
        raise NotImplementedError()

    @classmethod
    def parse_tokens(cls, block, text, state):
        if state.depth() >= block.max_nested_level - 1 and cls.name in block.rules:
            rules = list(block.rules)
            rules.remove(cls.name)
        else:
            rules = block.rules
        child = state.child_state(text)
        block.parse(child, rules)
        return child.tokens

    @staticmethod
    def parse_options(m: re.Match):
        text = m.group('options')
        if not text.strip():
            return []
        options = []
        for line in re.split('\\n+', text):
            line = line.strip()[1:]
            if not line:
                continue
            i = line.find(':')
            k = line[:i]
            v = line[i + 1:].strip()
            options.append((k, v))
        return options