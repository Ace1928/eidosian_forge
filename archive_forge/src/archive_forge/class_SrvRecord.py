import typing as t
class SrvRecord(t.NamedTuple):
    target: str
    port: int
    weight: int
    priority: int

    @classmethod
    def lookup(cls, service: str, proto: str, name: str) -> t.List['SrvRecord']:
        """Performs an SRV lookup.

        Args:
            service: The SRV service.
            proto: The SRV protocol.
            name: The SRV name.

        Returns:
            List[SrvRecord]: A list of records ordered by priority and weight.
        """
        record = f'_{service}._{proto}.{name}'
        answers: t.List[SrvRecord] = []
        for answer in dns.resolver.resolve(record, 'SRV'):
            answers.append(SrvRecord(target=str(answer.target), port=answer.port, weight=answer.weight, priority=answer.priority))
        return sorted(answers, key=lambda a: (a.priority, -a.weight))