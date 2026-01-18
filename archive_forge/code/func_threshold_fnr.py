def threshold_fnr(self, fnr):
    """Approximate the log-odds threshold which makes the type II error (false negative rate)."""
    i = -1
    prob = 0.0
    while prob < fnr:
        i += 1
        prob += self.mo_density[i]
    return self.min_score + i * self.step