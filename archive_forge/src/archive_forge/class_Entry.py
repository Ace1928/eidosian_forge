import collections
import urllib.parse
import urllib.request
class Entry:
    """An entry has one or more user-agents and zero or more rulelines"""

    def __init__(self):
        self.useragents = []
        self.rulelines = []
        self.delay = None
        self.req_rate = None

    def __str__(self):
        ret = []
        for agent in self.useragents:
            ret.append(f'User-agent: {agent}')
        if self.delay is not None:
            ret.append(f'Crawl-delay: {self.delay}')
        if self.req_rate is not None:
            rate = self.req_rate
            ret.append(f'Request-rate: {rate.requests}/{rate.seconds}')
        ret.extend(map(str, self.rulelines))
        return '\n'.join(ret)

    def applies_to(self, useragent):
        """check if this entry applies to the specified agent"""
        useragent = useragent.split('/')[0].lower()
        for agent in self.useragents:
            if agent == '*':
                return True
            agent = agent.lower()
            if agent in useragent:
                return True
        return False

    def allowance(self, filename):
        """Preconditions:
        - our agent applies to this entry
        - filename is URL decoded"""
        for line in self.rulelines:
            if line.applies_to(filename):
                return line.allowance
        return True