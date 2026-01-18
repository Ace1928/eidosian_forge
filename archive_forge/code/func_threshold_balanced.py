def threshold_balanced(self, rate_proportion=1.0, return_rate=False):
    """Approximate log-odds threshold making FNR equal to FPR times rate_proportion."""
    i = self.n_points
    fpr = 0.0
    fnr = 1.0
    while fpr * rate_proportion < fnr:
        i -= 1
        fpr += self.bg_density[i]
        fnr -= self.mo_density[i]
    if return_rate:
        return (self.min_score + i * self.step, fpr)
    else:
        return self.min_score + i * self.step