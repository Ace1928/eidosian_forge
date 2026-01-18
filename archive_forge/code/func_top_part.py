important invariant is that the parts on the stack are themselves in
def top_part(self):
    """Return current top part on the stack, as a slice of pstack.

        """
    return self.pstack[self.f[self.lpart]:self.f[self.lpart + 1]]