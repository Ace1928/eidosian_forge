from openunmix import utils
import torch.hub
def umxl_spec(targets=None, device='cpu', pretrained=True):
    from .model import OpenUnmix
    target_urls = {'bass': 'https://zenodo.org/records/5069601/files/bass-2ca1ce51.pth', 'drums': 'https://zenodo.org/records/5069601/files/drums-69e0ebd4.pth', 'other': 'https://zenodo.org/records/5069601/files/other-c8c5b3e6.pth', 'vocals': 'https://zenodo.org/records/5069601/files/vocals-bccbd9aa.pth'}
    if targets is None:
        targets = ['vocals', 'drums', 'bass', 'other']
    max_bin = utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000)
    target_models = {}
    for target in targets:
        target_unmix = OpenUnmix(nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=1024, max_bin=max_bin)
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(target_urls[target], map_location=device)
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()
        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models