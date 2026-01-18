def update_arrows(self):
    if self.in_arrow:
        self.in_arrow.vectorize()
    if self.out_arrow:
        self.out_arrow.vectorize()