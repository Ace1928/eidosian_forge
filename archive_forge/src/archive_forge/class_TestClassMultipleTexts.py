import cirq.testing
class TestClassMultipleTexts:

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('TestClass')
        else:
            p.text("I'm so pretty")
            p.text(' I am')