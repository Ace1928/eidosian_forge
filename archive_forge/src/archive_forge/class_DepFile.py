from __future__ import annotations
import typing as T
class DepFile:

    def __init__(self, lines: T.Iterable[str]):
        rules = parse(lines)
        depfile: T.Dict[str, Target] = {}
        for targets, deps in rules:
            for target in targets:
                t = depfile.setdefault(target, Target(deps=set()))
                for dep in deps:
                    t.deps.add(dep)
        self.depfile = depfile

    def get_all_dependencies(self, name: str, visited: T.Optional[T.Set[str]]=None) -> T.List[str]:
        deps: T.Set[str] = set()
        if not visited:
            visited = set()
        if name in visited:
            return []
        visited.add(name)
        target = self.depfile.get(name)
        if not target:
            return []
        deps.update(target.deps)
        for dep in target.deps:
            deps.update(self.get_all_dependencies(dep, visited))
        return sorted(deps)