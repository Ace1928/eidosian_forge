import os
from os.path import join
import shutil
import time
import uuid
from lion_pytorch import Lion
import pytest
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter
@pytest.mark.parametrize('optim_name', optimizer_names_32bit, ids=id_formatter('opt'))
@pytest.mark.parametrize('gtype', [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize('dim1', [1024], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [32, 1024, 4097, 1], ids=id_formatter('dim2'))
def test_optimizer32bit(dim1, dim2, gtype, optim_name):
    if gtype == torch.bfloat16 and optim_name in ['momentum', 'rmsprop']:
        pytest.skip()
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()
    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])
    if gtype == torch.float32:
        atol, rtol = (1e-06, 1e-05)
    elif gtype == torch.bfloat16:
        atol, rtol = (0.001, 0.01)
    else:
        atol, rtol = (0.0001, 0.001)
    for i in range(k):
        g = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()
        bnb_optimizer.step()
        torch_optimizer.step()
        for name1, name2 in str2statenames[optim_name]:
            torch.testing.assert_close(torch_optimizer.state[p1][name1], bnb_optimizer.state[p2][name2].cuda(), atol=atol, rtol=rtol)
        assert_most_approx_close(p1, p2.float(), atol=atol, rtol=rtol, max_error_count=10)
        if i % (k // 5) == 0 and i > 0:
            path = get_temp_dir()
            torch.save(bnb_optimizer.state_dict(), join(path, 'opt.pt'))
            del bnb_optimizer
            bnb_optimizer = None
            bnb_optimizer = str2optimizers[optim_name][1]([p2])
            bnb_optimizer.load_state_dict(torch.load(join(path, 'opt.pt')))
            rm_path(path)
            assert_most_approx_close(p1, p2.float(), atol=atol, rtol=rtol, max_error_count=10)
            for name1, name2 in str2statenames[optim_name]:
                assert_most_approx_close(torch_optimizer.state[p1][name1], bnb_optimizer.state[p2][name2], atol=atol, rtol=rtol, max_error_count=10)
        if gtype != torch.float32:
            p1.data = p1.data.to(p2.dtype).float()
            p2.copy_(p1.data)
            torch.testing.assert_close(p1.to(p2.dtype), p2)
        if optim_name in ['lars', 'lamb']:
            assert bnb_optimizer.state[p2]['unorm_vec'] > 0.0