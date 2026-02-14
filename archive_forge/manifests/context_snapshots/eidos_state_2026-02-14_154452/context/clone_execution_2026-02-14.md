# Clone Execution Report - 2026-02-14

## Legacy Migration Complete
- Source media migrated from previous `/dev/sda` layout before wipe.
- Curated targets:
  - `/home/lloyd/eidosian_forge/projects/legacy/evie`
  - `/home/lloyd/eidosian_forge/projects/legacy/indego_snake_game`
  - `/home/lloyd/eidosian_forge/projects/legacy/minecraft_ai_indego`
  - `/home/lloyd/eidosian_forge/projects/legacy/ai_timeline`
  - `/home/lloyd/eidosian_forge/projects/legacy/3dstuff`
  - `/home/lloyd/eidosian_forge/projects/research/universal_intelligence_potential`
  - `/home/lloyd/eidosian_forge/docs/legacy/aurora_archive`
  - `/home/lloyd/eidosian_forge/docs/legacy/aurora_chat_logs`
- Run directory:
  - `/home/lloyd/eidosian_forge/archive_forge/manifests/legacy_import_2026-02-14_123808`
- Latest summary:
  - `/home/lloyd/eidosian_forge/archive_forge/manifests/legacy_import_latest_summary.txt`

## Clone Kit Complete
- Profile export:
  - `/home/lloyd/eidosian_forge/archive_forge/clone_kits/host_profile_Eidos_2026-02-14_124522`
- Includes:
  - dpkg/snap/pip/ollama manifests
  - dconf + GNOME key settings
  - home config + fonts + backgrounds + pictures
  - encrypted secrets archive (`secrets.tar.enc`)

## USB Rebuild Complete (`/dev/sda`)
Partitioning + labels:
- `sda1` FAT32 `VENTOY_EFI` (2G)
- `sda2` exFAT `LIVE_MULTI` (126G)
- `sda3` LUKS2 -> ext4 `EIDOS_VAULT` (803.5G)

Boot assets staged:
- `LIVE_MULTI/ubuntu-live.iso` (Ubuntu 24.04.4 desktop)
- `LIVE_MULTI/systemrescue.iso` (SystemRescue 12.03)
- `LIVE_MULTI/casper-rw` (32G persistence)
- EFI GRUB config:
  - `/boot/grub/grub.cfg` on `VENTOY_EFI`

## Encrypted Vault Payload Complete
Stored under:
- `/mnt/eidos_vault/eidos_clone/` (while mounted)

Payload sections:
- `clone_kits/`
- `legacy_manifests/`
- `scripts/`
- `system_snapshots/` (`boot.tar`, `boot_efi.tar`, `etc.tar`)
- `checksums/sha256_manifest.txt`
- `checksums/size_summary.txt`

## Sync Mesh Setup Complete (local node)
- Installed:
  - `syncthing`
  - `tailscale`
  - `age`
- Services:
  - `tailscaled` enabled/active
  - `syncthing` user service enabled/active
- Mesh manifest:
  - `/home/lloyd/.config/eidos-sync/sync_mesh_manifest.v1.json`
- Syncthing ignore file:
  - `/home/lloyd/eidosian_forge/data/shared_state/.stignore`

## Pending User Action
- Tailscale auth still required to join tailnet:
  - `sudo tailscale up --hostname=eidos-laptop --accept-routes=false`
