from pythran.passmanager import ModuleAnalysis
def visit_body(self, body):
    body_as_tuple = tuple(body)
    self.result[body_as_tuple] = current = self.current
    self.current += (body_as_tuple,)
    for stmt in body:
        self.generic_visit(stmt)
    self.current = current